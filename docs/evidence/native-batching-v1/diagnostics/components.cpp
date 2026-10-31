#include "stealth/numeric.hpp"
#include "stealth/feed_forward.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <bit>
using namespace stealth::numeric;
using Clock=std::chrono::steady_clock;
void check(Error e){if(e!=Error::ok)throw std::runtime_error(error_name(e));}
void fill(std::vector<float>& x){for(std::size_t i=0;i<x.size();++i)x[i]=static_cast<float>(static_cast<int>((i*37+13)%257)-128)/127.1F;}
template<class A,class B>void measure(const char*name,A a,B b,std::vector<float>&x,std::vector<float>&y){
 for(int i=0;i<64;++i){a();b();}
 for(std::size_t i=0;i<x.size();++i)if(std::bit_cast<unsigned>(x[i])!=std::bit_cast<unsigned>(y[i]))throw std::runtime_error("Parity");
 std::vector<double>at,bt;
 auto timed=[](auto fn){auto t=Clock::now();fn();return std::chrono::duration<double,std::nano>(Clock::now()-t).count();};
 for(int i=0;i<2000;++i){if(i%2){at.push_back(timed(a));bt.push_back(timed(b));}else{bt.push_back(timed(b));at.push_back(timed(a));}}
 std::sort(at.begin(),at.end());std::sort(bt.begin(),bt.end());
 std::cout<<"{\"stage\":\""<<name<<"\",\"before_ns\":"<<at[999]<<",\"after_ns\":"<<bt[999]<<"}\n";
}
void dense(const char*name,std::size_t in,std::size_t out,std::size_t rows,std::size_t before,std::size_t after){
 std::vector<float>w(in*out),bias(out),x(rows*in),a(rows*out),b(rows*out),scratch(rows*out);fill(w);fill(bias);fill(x);Dense d(in,out,w,bias);
 auto run=[&](std::size_t tile,std::vector<float>&output){for(std::size_t r=0;r<rows;r+=tile){auto n=std::min(tile,rows-r);check(d.forward_rows(std::span(x).subspan(r*in,n*in),n,std::span(output).subspan(r*out,n*out),scratch));}};
 measure(name,[&]{run(before,a);},[&]{run(after,b);},a,b);
}
int main(){
 dense("embed 1 to 32",7,64,96,1,32);dense("qkv 4 to 32",64,192,96,4,32);dense("qkv 4 to 8",64,192,96,4,8);
 dense("project 1 to 4",64,64,4,1,4);dense("head 1 to 4",64,1,4,1,4);
 {std::vector<float>x(96*64),scale(64,1),shift(64,0),a(x.size()),b(x.size()),scratch(x.size());fill(x);
 measure("norm 1 to 32",[&]{for(int r=0;r<96;++r)check(layer_norm(std::span(x).subspan(r*64,64),scale,shift,1e-5F,std::span(a).subspan(r*64,64),scratch));},[&]{for(int r=0;r<96;r+=32)check(layer_norm_rows(std::span(x).subspan(r*64,32*64),32,scale,shift,1e-5F,std::span(b).subspan(r*64,32*64),scratch));},a,b);}
 {std::vector<float>x(4*64),scale(64,1),shift(64,0),up(64*256),ub(256),down(256*64),db(64),a(x.size()),b(x.size());fill(x);fill(up);fill(down);FeedForward d(64,256,{scale,shift,up,ub,down,db});FeedForwardWorkspace ws(64,256,4);
 measure("MLP 1 to 4",[&]{for(int r=0;r<4;++r)check(d.forward(std::span(x).subspan(r*64,64),std::span(a).subspan(r*64,64),ws));},[&]{check(d.forward_rows(x,4,b,ws));},a,b);}
}
