import os

def merge_files(file1, file2, output_file):
    # 检查文件是否存在
    if not os.path.exists(file1):
        print(f"文件 {file1} 不存在，请检查路径！")
        return
    if not os.path.exists(file2):
        print(f"文件 {file2} 不存在，请检查路径！")
        return

    # 打开文件1和文件2并将其内容写入到输出文件中
    with open(output_file, 'w') as outfile:
        # 读取文件1内容并写入
        with open(file1, 'r') as f1:
            outfile.writelines(f1.readlines())
        # 读取文件2内容并写入
        with open(file2, 'r') as f2:
            outfile.writelines(f2.readlines())

    print(f"文件已成功合并并保存为 {output_file}")

# 使用该函数
file1 = 's1-weak-split.lst'
file2 = 'flood_train_data.lst'
output_file = 'merged_file.lst'
merge_files(file1, file2, output_file)
