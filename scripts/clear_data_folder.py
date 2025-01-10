import os
import schedule
import time


def delete_files_in_folder(folder_path):
    # 遍历目录及子目录
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


# 定义定期任务
def job():
    folder_path = "../data/logs"  # 指定要删除文件的文件夹路径
    delete_files_in_folder(folder_path)


# 每天午夜 00:00 执行删除操作
schedule.every().day.at("00:00").do(job)

# 保持脚本运行，执行定时任务
while True:
    schedule.run_pending()
    time.sleep(1)
