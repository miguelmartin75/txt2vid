#include <iostream>
using namespace std;

using ll = long long;
const int MAX_N = 10000000;
const ll M = 1e9 + 7;

ll add(ll x, ll y)
{
    return ((x % M) + (y % M)) % M;
}

// won't overflow since it is only 2*M worst-case
ll mult(ll x, ll y)
{
    return ((x % M) * (y % M)) % M;
}

ll dp[MAX_N + 1][2];
ll solve(ll n, bool b)
{
    if(n == 1) return !b;
    if(dp[n][b] != -1) 
        return dp[n][b];
    dp[n][b] = add(solve(n - 1, !b), mult(2, solve(n - 1, false)));
    assert(dp[n][b] < M);
    return dp[n][b];
}

int main(int argc, char *argv[])
{
    int n;
    cin >> n;

    for(int i = 0; i <= n; ++i) 
        for(int b = 0; b < 2; ++b) dp[i][b] = -1;

    cout << solve(n, true) << endl;
}
