#ifndef  LINEAR_SUM_ASSIGNMENT_H
#define  LINEAR_SUM_ASSIGNMENT_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// 定义常量用于错误代码
#define RECTANGULAR_LSAP_INFEASIBLE -1
#define RECTANGULAR_LSAP_INVALID -2
template <typename T> std::vector<intptr_t> argsort_iter(const std::vector<T> &v)
{
    std::vector<intptr_t> index(v.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&v](intptr_t i, intptr_t j)
              {return v[i] < v[j];});
    return index;
}
// 辅助函数：找到增广路径
intptr_t augmenting_path(intptr_t nc, double *cost, std::vector<double>& u,
                std::vector<double>& v, std::vector<intptr_t>& path,
                std::vector<intptr_t>& row4col,
                std::vector<double>& shortestPathCosts, intptr_t i,
                std::vector<bool>& SR, std::vector<bool>& SC,
                std::vector<intptr_t>& remaining, double* p_minVal)
{
    double minVal = 0;

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes.  Here we use a vector, as it is more efficient in C++.
    intptr_t num_remaining = nc;
    for (intptr_t it = 0; it < nc; it++) {
        // Filling this up in reverse order ensures that the solution of a
        // constant cost matrix is the identity matrix (c.f. #11602).
        remaining[it] = nc - it - 1;
    }

    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);
    std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), INFINITY);

    // find shortest augmenting path
    intptr_t sink = -1;
    while (sink == -1) {

        intptr_t index = -1;
        double lowest = INFINITY;
        SR[i] = true;

        for (intptr_t it = 0; it < num_remaining; it++) {
            intptr_t j = remaining[it];

            double r = minVal + cost[i * nc + j] - u[i] - v[j];
            if (r < shortestPathCosts[j]) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }

            // When multiple nodes have the minimum cost, we select one which
            // gives us a new sink node. This is particularly important for
            // integer cost matrices with small co-efficients.
            if (shortestPathCosts[j] < lowest ||
                (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        if (minVal == INFINITY) { // infeasible cost matrix
            return -1;
        }

        intptr_t j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
    }

    *p_minVal = minVal;
    return sink;
}

// 线性分配问题求解函数
int solve(intptr_t nr, intptr_t nc, double* cost, bool maximize,
      int64_t* a, int64_t* b)
{
    // handle trivial inputs
    if (nr == 0 || nc == 0) {
        return 0;
    }

    // tall rectangular cost matrix must be transposed
    bool transpose = nc < nr;

    // make a copy of the cost matrix if we need to modify it
    std::vector<double> temp;
    if (transpose || maximize) {
        temp.resize(nr * nc);

        if (transpose) {
            for (intptr_t i = 0; i < nr; i++) {
                for (intptr_t j = 0; j < nc; j++) {
                    temp[j * nr + i] = cost[i * nc + j];
                }
            }

            std::swap(nr, nc);
        }
        else {
            std::copy(cost, cost + nr * nc, temp.begin());
        }

        // negate cost matrix for maximization
        if (maximize) {
            for (intptr_t i = 0; i < nr * nc; i++) {
                temp[i] = -temp[i];
            }
        }

        cost = temp.data();
    }

    // test for NaN and -inf entries
    for (intptr_t i = 0; i < nr * nc; i++) {
        if (cost[i] != cost[i] || cost[i] == -INFINITY) {
            return RECTANGULAR_LSAP_INVALID;
        }
    }

    // initialize variables
    std::vector<double> u(nr, 0);
    std::vector<double> v(nc, 0);
    std::vector<double> shortestPathCosts(nc);
    std::vector<intptr_t> path(nc, -1);
    std::vector<intptr_t> col4row(nr, -1);
    std::vector<intptr_t> row4col(nc, -1);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::vector<intptr_t> remaining(nc);

    // iteratively build the solution
    for (intptr_t curRow = 0; curRow < nr; curRow++) {

        double minVal;
        intptr_t sink = augmenting_path(nc, cost, u, v, path, row4col,
                                        shortestPathCosts, curRow, SR, SC,
                                        remaining, &minVal);
        if (sink < 0) {
            return RECTANGULAR_LSAP_INFEASIBLE;
        }

        // update dual variables
        u[curRow] += minVal;
        for (intptr_t i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                u[i] += minVal - shortestPathCosts[col4row[i]];
            }
        }

        for (intptr_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= minVal - shortestPathCosts[j];
            }
        }

        // augment previous solution
        intptr_t j = sink;
        while (1) {
            intptr_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) {
                break;
            }
        }
    }

    if (transpose) {
        intptr_t i = 0;
        for (auto v: argsort_iter(col4row)) {
            a[i] = col4row[v];
            b[i] = v;
            i++;
            // cout<<"v:" <<i<<" "<<col4row[v]<<" "<<v<<endl;
        }
        // if (i<nc)
        // {

        // }
        // cout<<"v:" <<i<<endl; 
    }
    else {
        for (intptr_t i = 0; i < nr; i++) {
            a[i] = i;
            b[i] = col4row[i];
        }
    }
    // cout<<"nc"<<nc<<" nr"<<nr<<endl;
    // cout<<"a"<<malloc_usable_size(a) / sizeof(*a)<<" b"<<malloc_usable_size(b) / sizeof(*b)<<endl;
    
    // if (transpose)
    // {
    //     int64_t* newArr = new int64_t[nr];
    //     // 复制原数组的前newSize个元素到新的数组中
    //     for (int i =0 ;i<nr;i++)
    //         newArr[i] = a[i];
    //     // std::memcpy(newArr, a, nr * sizeof(int64_t));
    //     // 释放原数组的内存
    //     delete[] a;
    //     // 更新指针和大小
    //     a = newArr;

    //     int64_t* newBrr = new int64_t[nr];
    //     // 复制原数组的前newSize个元素到新的数组中
    //     for (int i =0 ;i<nr;i++)
    //         newBrr[i] = b[i];
    //     // 释放原数组的内存
    //     delete[] b;
    //     // 更新指针和大小
    //     b = newBrr;
    //     }
    return 0;
}

// 线性分配问题求解函数（外部接口）
int linear_sum_assignment(double* cost_matrix, int num_rows, int num_cols, bool maximize, int64_t** a, int64_t** b) {
    // 分配结果数组
    // int size_ab = num_rows > num_cols ? num_rows : num_cols ;
    int64_t* a_result = new int64_t[num_cols];
    int64_t* b_result = new int64_t[num_cols];
    if (!a_result || !b_result) {
        delete[] a_result;
        delete[] b_result;
        return -1; // 内存分配失败
    }

    // 调用求解函数
    int ret = solve(num_rows, num_cols, cost_matrix, maximize, a_result, b_result);

    if (ret == RECTANGULAR_LSAP_INFEASIBLE) {
        delete[] a_result;
        delete[] b_result;
        return RECTANGULAR_LSAP_INFEASIBLE; // 成本矩阵不可行
    } else if (ret == RECTANGULAR_LSAP_INVALID) {
        delete[] a_result;
        delete[] b_result;
        return RECTANGULAR_LSAP_INVALID; // 矩阵包含无效的数值
    }

    *a = a_result;
    *b = b_result;
    return 0; // 成功
}

#endif