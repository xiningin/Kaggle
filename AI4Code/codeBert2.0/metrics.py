from bisect import bisect #用于维护有序list，使用二分查找法插入适当位置

#计算修改到正确顺序需要交换的次数
def count_inversions(ranks):
    inversions = 0
    sorted_so_far = []
    for i,u in enumerate(ranks):
        j = bisect(sorted_so_far , u)
        inversions += i - j
        sorted_so_far.insert(j,u)
    return inversions

#计算损失的函数
def kendall_tau(groud_truth , predication):
    total_inversions = 0
    total_2max = 0
    for gt , pred in zip(groud_truth , predication):
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n -1 )
    return 1 - 4 * total_inversions / total_2max #题目的score计算方式