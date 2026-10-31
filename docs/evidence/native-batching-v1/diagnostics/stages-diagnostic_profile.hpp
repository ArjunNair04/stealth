#pragma once
#include <chrono>
#include <cstdio>
#include <cstdint>
namespace stealth_diag {
inline std::uint64_t windows=0;
inline double stages[6]{};
inline auto now(){return std::chrono::steady_clock::now();}
inline void account(int slot,std::chrono::steady_clock::time_point& start){auto end=now();stages[slot]+=std::chrono::duration<double,std::nano>(end-start).count();start=end;}
struct Report{~Report(){std::fprintf(stderr,"{\"windows\":%llu,\"ns_per_window\":[",static_cast<unsigned long long>(windows));for(int i=0;i<6;++i)std::fprintf(stderr,"%s%.9f",i?",":"",stages[i]/static_cast<double>(windows));std::fprintf(stderr,"]}\n");}};
inline Report report;
}
