// Exhaustive 3x3 expectimax. No horizon, sampling, or heuristic leaf values.
// Build: c++ -O3 -std=c++17 -shared -fPIC solver.cpp -o solver.so
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Board = std::uint64_t;  // Nine four-bit tile exponents, empty = 0.
struct Entry { Board key = 0; double value = 0; };

class Table {
public:
    std::vector<Entry> slots = std::vector<Entry>(1024);
    std::uint64_t count = 0;

    static std::uint64_t hash(Board key) {
        key ^= key >> 30;
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= key >> 27;
        key *= 0x94d049bb133111ebULL;
        return key ^ (key >> 31);
    }
    bool get(Board key, double& value) const {
        auto index = hash(key) & (slots.size() - 1);
        while (slots[index].key) {
            if (slots[index].key == key) {
                value = slots[index].value;
                return true;
            }
            index = (index + 1) & (slots.size() - 1);
        }
        return false;
    }
    void put(Board key, double value) {
        if ((count + 1) * 4 > slots.size() * 3) {
            auto previous = std::move(slots);
            slots = std::vector<Entry>(previous.size() * 2);
            count = 0;
            for (const auto& entry : previous)
                if (entry.key) put(entry.key, entry.value);
        }
        auto index = hash(key) & (slots.size() - 1);
        while (slots[index].key && slots[index].key != key)
            index = (index + 1) & (slots.size() - 1);
        if (!slots[index].key) ++count;
        slots[index] = {key, value};
    }
};

Board transpose(Board b) {
    return (b & 0xf000f000fULL)
        | ((b & 0x000f000f0ULL) << 8)
        | ((b & 0x0f000f000ULL) >> 8)
        | ((b & 0x000000f00ULL) << 16)
        | ((b & 0x00f000000ULL) >> 16);
}

Board reflect(Board b) {
    return (b & 0x0f00f00f0ULL)
        | ((b & 0x00f00f00fULL) << 8)
        | ((b & 0xf00f00f00ULL) >> 8);
}

Board flip(Board b) {
    return ((b & 0xfffULL) << 24) | (b & 0xfff000ULL) | (b >> 24);
}

Board canonical(Board b) {
    const Board r = reflect(b), f = flip(b), rf = reflect(f);
    return std::min({b, r, f, rf, transpose(b), transpose(r),
                     transpose(f), transpose(rf)});
}

struct Header {
    char magic[8] = {'E','M','3','S','U','M','0','1'};
    std::uint64_t target = 0;
    double bonus = 0;
    std::uint64_t capacity = 0, count = 0;
};

class Solver {
public:
    int target;
    double bonus;
    Table players, chances;
    std::array<Board, 4096> left{}, right{};
    std::string path;
    bool prepared = false;
    std::uint64_t next_report = 5000000;

    Solver(int target_, double bonus_, const char* path_)
        : target(target_), bonus(bonus_), path(path_) {
        for (unsigned row = 0; row < 4096; ++row) {
            unsigned tiles[3], length = 0, output[3] = {}, out = 0;
            for (int i = 0; i < 3; ++i)
                if (unsigned tile = (row >> (4 * i)) & 15) tiles[length++] = tile;
            for (unsigned i = 0; i < length; ++i) {
                if (i + 1 < length && tiles[i] == tiles[i + 1]) {
                    output[out++] = tiles[i] + 1;
                    ++i;
                } else output[out++] = tiles[i];
            }
            left[row] = output[0] | (output[1] << 4) | (output[2] << 8);
        }
        for (unsigned row = 0; row < 4096; ++row) {
            unsigned reversed = ((row & 15) << 8) | (row & 240) | (row >> 8);
            Board moved = left[reversed];
            right[row] = ((moved & 15) << 8) | (moved & 240) | (moved >> 8);
        }
    }

    Board move(Board b, int action) const {
        if (action >= 2) b = transpose(b);
        const auto& rows = (action & 1) ? right : left;
        Board result = rows[b & 4095] | (rows[(b >> 12) & 4095] << 12)
            | (rows[b >> 24] << 24);
        return action >= 2 ? transpose(result) : result;
    }

    double player(Board b) {
        b = canonical(b);
        double value;
        if (players.get(b, value)) return value;
        value = -std::numeric_limits<double>::infinity();
        for (int action = 0; action < 4; ++action) {
            Board moved = move(b, action);
            if (moved != b) value = std::max(value, chance(moved));
        }
        if (!std::isfinite(value)) {
            int sum = 0, maximum = 0;
            for (int i = 0; i < 9; ++i) {
                int exponent = (b >> (4 * i)) & 15;
                int tile = exponent ? 1 << exponent : 0;
                sum += tile;
                maximum = std::max(maximum, tile);
            }
            value = sum + (maximum >= target ? bonus : 0);
        }
        players.put(b, value);
        if (!prepared && players.count >= next_report) {
            std::fprintf(stderr, "Expectimax: solved %llu player states\n",
                         static_cast<unsigned long long>(players.count));
            next_report += 5000000;
        }
        return value;
    }

    double chance(Board b) {
        b = canonical(b);
        double value;
        if (chances.get(b, value)) return value;
        value = 0;
        int empty = 0;
        for (int i = 0; i < 9; ++i) {
            if ((b >> (4 * i)) & 15) continue;
            ++empty;
            value += 0.9 * player(b | (Board(1) << (4 * i)))
                   + 0.1 * player(b | (Board(2) << (4 * i)));
        }
        if (!empty) throw std::runtime_error("chance node has no empty cell");
        value /= empty;
        chances.put(b, value);
        return value;
    }

    bool load() {
        if (path.empty() || !std::filesystem::exists(path)) return false;
        std::ifstream input(path, std::ios::binary);
        Header header, expected;
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input || std::memcmp(header.magic, expected.magic, 8)
            || header.target != static_cast<unsigned>(target) || header.bonus != bonus
            || header.capacity < 1024 || header.capacity > (1ULL << 27)
            || (header.capacity & (header.capacity - 1)) || header.count > header.capacity
            || std::filesystem::file_size(path) != sizeof(header) + header.capacity * sizeof(Entry))
            throw std::runtime_error("invalid expectimax cache; remove " + path);
        players.slots.resize(header.capacity);
        input.read(reinterpret_cast<char*>(players.slots.data()),
                   header.capacity * sizeof(Entry));
        if (!input) throw std::runtime_error("cannot read expectimax cache " + path);
        players.count = header.count;
        prepared = true;
        return true;
    }

    void prepare() {
        if (prepared || load()) return;
        std::fprintf(stderr, "Expectimax: solving all 3x3 initial positions to terminal states...\n");
        // The initial board has two independent 2/4 drops in distinct cells.
        for (int i = 0; i < 9; ++i)
            for (int j = i + 1; j < 9; ++j)
                for (int a = 1; a <= 2; ++a)
                    for (int b = 1; b <= 2; ++b)
                        player((Board(a) << (4 * i)) | (Board(b) << (4 * j)));
        std::fprintf(stderr, "Expectimax: complete, %llu player / %llu chance states\n",
                     static_cast<unsigned long long>(players.count),
                     static_cast<unsigned long long>(chances.count));
        prepared = true;
        if (!path.empty()) {
            Header header;
            header.target = target;
            header.bonus = bonus;
            header.capacity = players.slots.size();
            header.count = players.count;
            std::string temporary = path + ".tmp";
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(&header), sizeof(header));
            output.write(reinterpret_cast<const char*>(players.slots.data()),
                         players.slots.size() * sizeof(Entry));
            output.close();
            if (!output) throw std::runtime_error("cannot write expectimax cache " + path);
            std::filesystem::rename(temporary, path);
        }
        // Subsequent queries average cached player values; no chance table needed.
        chances = Table();
    }
};

void error_text(char* error, const std::exception& exception) {
    std::snprintf(error, 1024, "%s", exception.what());
}
}  // namespace

extern "C" {
void* em_create(int target, double bonus, const char* path, char* error) {
    try { return new Solver(target, bonus, path); }
    catch (const std::exception& exception) { error_text(error, exception); return nullptr; }
}
void em_destroy(void* solver) { delete static_cast<Solver*>(solver); }
int em_prepare(void* solver, char* error) {
    try { static_cast<Solver*>(solver)->prepare(); return 0; }
    catch (const std::exception& exception) { error_text(error, exception); return -1; }
}
int em_values(void* solver, Board board, double* values, char* error) {
    try {
        auto& engine = *static_cast<Solver*>(solver);
        for (int action = 0; action < 4; ++action) {
            Board moved = engine.move(board, action);
            values[action] = moved == board ? -std::numeric_limits<double>::infinity()
                                           : engine.chance(moved);
        }
        return 0;
    } catch (const std::exception& exception) { error_text(error, exception); return -1; }
}
Board em_move(void* solver, Board board, int action) {
    return static_cast<Solver*>(solver)->move(board, action);
}
}
