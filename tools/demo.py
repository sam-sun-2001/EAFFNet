import torch
def main():
    # 模型的输出概率分布
    import torch

    # 假设输入张量
    input_tensor = torch.tensor([
        [0.1, 0.2, 0.3],#在这个例子中，index_tensor 指定了每行中要抓取的位置
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

    # 假设索引张量，表示要抓取的位置
    index_tensor = torch.tensor([
        [2, 0, 1],
        [1, 2, 0],
        [0, 2, 1]
    ])

    # 使用 gather 进行值的抓取
    selected_values = torch.gather(input_tensor, 1, index_tensor)

    # 打印结果
    print("输入张量：")
    print(input_tensor)
    print("\n索引张量：")
    print(index_tensor)
    print("\n抓取的值：")
    print(selected_values)


if __name__ == '__main__':
    main()
