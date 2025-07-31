import json
import gzip

def read_first_jsonl_entry(file_path):
    """读取JSONL文件的第一条数据并返回"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取第一行
            first_line = f.readline()
            if not first_line:
                return "文件为空"
            
            # 解析JSON
            first_entry = json.loads(first_line)
            return first_entry
    except FileNotFoundError:
        return f"错误：文件 '{file_path}' 不存在"
    except json.JSONDecodeError:
        return "错误：第一条数据不是有效的JSON格式"
    except Exception as e:
        return f"发生错误：{str(e)}"


def convert_gz_to_jsonl(gz_file_path, output_jsonl_path):
    """
    将jsonl.gz压缩文件转换为普通jsonl文件
    
    参数:
        gz_file_path: 输入的jsonl.gz文件路径
        output_jsonl_path: 输出的jsonl文件路径
    """
    try:
        # 打开压缩文件进行读取
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as gz_file:
            # 打开输出文件进行写入
            with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                # 逐行读取并写入
                for line in gz_file:
                    # 保留原始格式，直接写入
                    jsonl_file.write(line)
        
        print(f"成功转换：{gz_file_path} -> {output_jsonl_path}")
        return True
    
    except FileNotFoundError:
        print(f"错误：输入文件 '{gz_file_path}' 不存在")
    except Exception as e:
        print(f"转换失败：{str(e)}")
    return False

# 使用示例
# 使用示例
if __name__ == "__main__":
    # 输入的gz文件路径
    input_gz_path = "./minhash_results/deduplicated_output/00000.jsonl.gz"
    # 输出的jsonl文件路径
    output_jsonl_path = "../data/00000.jsonl"
    
    # 执行转换
    convert_gz_to_jsonl(input_gz_path, output_jsonl_path)