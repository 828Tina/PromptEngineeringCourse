from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


# 配置Minhash参数
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)

# 本地路径配置
LOCAL_BASE_PATH = "./minhash_results"
LOCAL_LOGS_FOLDER = "./minhash_logs"

# 输入数据路径
INPUT_PATH = "../data/pretrain.jsonl"

# 任务数量
TOTAL_TASKS = 10


# 主程序逻辑，放在if __name__ == '__main__'块中
if __name__ == '__main__':
    # 在Windows系统上可能需要这行代码，其他系统可以省略
    # from multiprocess import freeze_support
    # freeze_support()
    
    # 阶段1：计算每条数据的Minhash签名
    stage1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(INPUT_PATH),
            MinhashDedupSignature(
                output_folder=f"{LOCAL_BASE_PATH}/signatures",
                config=minhash_config,
                language=Languages.english
            ),
        ],
        tasks=TOTAL_TASKS,
        workers=5,
        logging_dir=f"{LOCAL_LOGS_FOLDER}/signatures",
    )
    
    
    # 阶段2：在每个桶中匹配签名
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{LOCAL_BASE_PATH}/signatures",
                output_folder=f"{LOCAL_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=5,
        logging_dir=f"{LOCAL_LOGS_FOLDER}/buckets",
    )
    
    
    # 阶段3：根据桶匹配结果聚类重复数据
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{LOCAL_BASE_PATH}/buckets",
                output_folder=f"{LOCAL_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=f"{LOCAL_LOGS_FOLDER}/clusters",
    )
    
    
    # 阶段4：过滤重复数据并输出结果
    stage4 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=INPUT_PATH,
                text_key="generated_text"),
            TokensCounter(),
            MinhashDedupFilter(
                input_folder=f"{LOCAL_BASE_PATH}/remove_ids",
                exclusion_writer=JsonlWriter(f"{LOCAL_BASE_PATH}/removed"),
            ),
            JsonlWriter(output_folder=f"{LOCAL_BASE_PATH}/deduplicated_output"),
        ],
        tasks=TOTAL_TASKS,
        workers=5,
        logging_dir=f"{LOCAL_LOGS_FOLDER}/filter",
    )
    
    # 设置依赖关系
    stage2.depends = stage1
    stage3.depends = stage2
    stage4.depends = stage3
    
    # 运行整个流水线
    stage4.run()
