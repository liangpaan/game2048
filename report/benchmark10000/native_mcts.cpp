// Evaluation-only acceleration of the existing Python MCTS. Same move order,
// tie breaking, tree policy, rollout, heuristic and CPython random stream.
// Build: g++ -O3 -std=c++17 -shared -fPIC native_mcts.cpp -o native_mcts.so
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>
using B=std::uint64_t;
struct PythonRandom {
    std::array<std::uint32_t,624> mt{}; int index=624;
    std::uint32_t next() {
        if(index>=624) {
            for(int i=0;i<624;i++) {
                auto y=(mt[i]&0x80000000U)|(mt[(i+1)%624]&0x7fffffffU);
                mt[i]=mt[(i+397)%624]^(y>>1)^((y&1)?0x9908b0dfU:0);
            }
            index=0;
        }
        auto y=mt[index++]; y^=y>>11; y^=(y<<7)&0x9d2c5680U;
        y^=(y<<15)&0xefc60000U; y^=y>>18; return y;
    }
    double uniform() {
        auto a=next()>>5,b=next()>>6;
        return (a*67108864.0+b)/9007199254740992.0;
    }
    int below(int n) {
        int bits=0;for(int v=n;v;v>>=1)++bits;
        unsigned r;do{r=next()>>(32-bits);}while(r>=unsigned(n));return r;
    }
};
struct Stats {int visits=0;double sum=0;};
struct Node {int visits=0,mask=0;std::array<Stats,4> stats{};};
struct Engine {
    int simulations,depth;double exploration,rollout_random,bonus,empty_weight,order_weight;
    PythonRandom rng;
    std::array<std::array<unsigned,4096>,4> lines{};
    std::array<std::array<std::array<double,4096>,3>,8> snake{};
    std::array<int,4096> sums{},empties{},maximum{};
    Engine(int s,int d,double c,double r,double e,double o,double t,const double* weights)
        :simulations(s),depth(d),exploration(c),rollout_random(r),bonus(t),empty_weight(e),order_weight(o) {
        for(unsigned row=0;row<4096;row++) {
            unsigned cells[3]={},n=0,out[3]={},j=0;
            for(int k=0;k<3;k++)if((row>>(4*k))&15)cells[n++]=(row>>(4*k))&15;
            for(unsigned i=0;i<n;i++) {
                if(i+1<n&&cells[i]==cells[i+1]){out[j++]=cells[i]+1;++i;}
                else out[j++]=cells[i];
            }
            lines[0][row]=out[0]|(out[1]<<4)|(out[2]<<8);
            for(int k=0;k<3;k++) {
                int rank=(row>>(4*k))&15;int value=rank?1<<rank:0;
                sums[row]+=value;empties[row]+=!rank;
                if(value>maximum[row])maximum[row]=value;
                for(int orient=0;orient<8;orient++)for(int rr=0;rr<3;rr++)
                    snake[orient][rr][row]+=value*weights[orient*9+rr*3+k];
            }
        }
        for(unsigned row=0;row<4096;row++) {
            auto rev=((row&15)<<8)|(row&240)|(row>>8),a=lines[0][rev];
            lines[1][row]=((a&15)<<8)|(a&240)|(a>>8);
        }
    }
    B transpose(B b) const {
        return (b&0xf000f000fULL)|((b&0x000f000f0ULL)<<8)|((b&0x0f000f000ULL)>>8)
             |((b&0x000000f00ULL)<<16)|((b&0x00f000000ULL)>>16);
    }
    B move(B b,int action) const {
        if(action>=2)b=transpose(b);
        auto& a=lines[action&1];
        B result=B(a[b&4095])|(B(a[(b>>12)&4095])<<12)|(B(a[b>>24])<<24);
        return action>=2?transpose(result):result;
    }
    int legal(B b)const {int mask=0;for(int a=0;a<4;a++)if(move(b,a)!=b)mask|=1<<a;return mask;}
    double heuristic(B b) const {
        int a=b&4095,c=(b>>12)&4095,d=b>>24;
        double ordered=0;
        for(auto& pattern:snake) {
            double v=pattern[0][a]+pattern[1][c]+pattern[2][d];
            if(v>ordered)ordered=v;
        }
        return sums[a]+sums[c]+sums[d]+empty_weight*(empties[a]+empties[c]+empties[d])
             +order_weight*ordered+bonus*(maximum[a]>=256||maximum[c]>=256||maximum[d]>=256);
    }
    B spawn(B b) {
        int positions[9],n=0;for(int i=0;i<9;i++)if(!((b>>(4*i))&15))positions[n++]=i;
        // Python evaluates the assignment RHS (tile draw) before the index
        // expression rng.choice(empty) in result[rng.choice(empty)] = tile.
        int rank=rng.uniform()<0.9?1:2;int pos=positions[rng.below(n)];
        return b|(B(rank)<<(4*pos));
    }
    B rollout(B b,int remaining) {
        for(int k=0;k<remaining;k++) {
            int mask=legal(b);if(!mask)break;
            int a=-1;
            if(rng.uniform()<rollout_random) {
                int actions[4],n=0;for(int i=0;i<4;i++)if(mask&(1<<i))actions[n++]=i;
                a=actions[rng.below(n)];
            } else {
                double best=-std::numeric_limits<double>::infinity();
                for(int i=0;i<4;i++)if(mask&(1<<i)) {
                    double v=heuristic(move(b,i));if(v>best){best=v;a=i;}
                }
            }
            b=spawn(move(b,a));
        }
        return b;
    }
    int choose(B b) {
        int root_mask=legal(b);if(!root_mask)return -1;
        if(!(root_mask&(root_mask-1)))for(int a=0;a<4;a++)if(root_mask&(1<<a))return a;
        std::vector<Node> nodes;nodes.reserve(simulations*depth+1);
        nodes.push_back(Node{0,root_mask,{}});
        std::unordered_map<B,int> ids;ids.reserve(simulations*depth+1);ids[b]=0;
        for(int sim=0;sim<simulations;sim++) {
            B state=b;std::vector<std::pair<int,int>> path;path.reserve(depth);
            for(int level=0;level<depth;level++) {
                auto it=ids.find(state);int id;
                if(it==ids.end()){id=nodes.size();ids[state]=id;nodes.push_back(Node{0,legal(state),{}});}
                else id=it->second;
                auto& node=nodes[id];if(!node.mask)break;
                int unvisited=0;for(int a=0;a<4;a++)if((node.mask&(1<<a))&&!node.stats[a].visits)unvisited|=1<<a;
                bool expanded=unvisited;int action=-1;
                double best=-std::numeric_limits<double>::infinity();
                double log_visits=expanded?0:std::log(node.visits+1.0);
                for(int a=0;a<4;a++)if((expanded?unvisited:node.mask)&(1<<a)) {
                    auto& stat=node.stats[a];
                    double v=expanded?heuristic(move(state,a)):
                        stat.sum/stat.visits+exploration*std::sqrt(log_visits/stat.visits);
                    if(v>best){best=v;action=a;}
                }
                path.emplace_back(id,action);state=spawn(move(state,action));
                if(expanded){state=rollout(state,depth-level-1);break;}
            }
            double value=heuristic(state);if(!legal(state))value-=256;
            for(auto [id,a]:path){nodes[id].visits++;nodes[id].stats[a].visits++;nodes[id].stats[a].sum+=value;}
        }
        int action=-1;double best=-std::numeric_limits<double>::infinity();
        for(int a=0;a<4;a++)if(nodes[0].stats[a].visits) {
            auto& stat=nodes[0].stats[a];double v=stat.sum/stat.visits;
            if(v>best){best=v;action=a;}
        }
        return action;
    }
};
extern "C" {
void* mc_create(int s,int d,double c,double r,double e,double o,double t,const double* weights){return new Engine(s,d,c,r,e,o,t,weights);}
void mc_destroy(void* p){delete static_cast<Engine*>(p);}
void mc_setstate(void* p,const std::uint32_t* state){auto& r=static_cast<Engine*>(p)->rng;for(int i=0;i<624;i++)r.mt[i]=state[i];r.index=state[624];}
void mc_getstate(void* p,std::uint32_t* state){auto& r=static_cast<Engine*>(p)->rng;for(int i=0;i<624;i++)state[i]=r.mt[i];state[624]=r.index;}
int mc_choose(void* p,B board){return static_cast<Engine*>(p)->choose(board);}
double mc_uniform(void* p){return static_cast<Engine*>(p)->rng.uniform();}
int mc_below(void* p,int n){return static_cast<Engine*>(p)->rng.below(n);}
B mc_move(void* p,B b,int a){return static_cast<Engine*>(p)->move(b,a);}
double mc_heuristic(void* p,B b){return static_cast<Engine*>(p)->heuristic(b);}
}
