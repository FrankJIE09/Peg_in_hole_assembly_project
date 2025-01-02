#!/bin/bash

# 检查是否在 Git 仓库目录中
if [ -d ".git" ]; then
  echo "Git 仓库已初始化。跳过 'git init' 步骤。"
else
  echo "初始化 Git 仓库..."
  git init
fi

# 创建或更新 .gitignore 文件
GITIGNORE_FILE=".gitignore"
IGNORE_CONTENT="

# 忽略日志文件
*.log

# 忽略临时文件
*.tmp
*.swp
*.bak

# 忽略操作系统生成的文件
.DS_Store
Thumbs.db

# 忽略编译生成的文件
*.o
*.obj
*.class
*.pyc
*.pyo
*.exe

# 忽略依赖文件夹
node_modules/
vendor/
__pycache__/

# 忽略环境文件
.env
.env.local
*.env

# 忽略 IDE 和编辑器配置文件
.idea/
.vscode/
*.sublime-*

"

echo "配置 .gitignore 文件..."
if [ ! -f "$GITIGNORE_FILE" ]; then
  touch "$GITIGNORE_FILE"
fi

# 将忽略规则添加到 .gitignore 文件（避免重复添加）
echo "$IGNORE_CONTENT" | while read -r line; do
  if ! grep -Fxq "$line" "$GITIGNORE_FILE"; then
    echo "$line" >> "$GITIGNORE_FILE"
  fi
done

echo ".gitignore 文件已配置完成！"

# 配置 pre-commit 钩子
PRE_COMMIT_FILE=".git/hooks/pre-commit"
echo "配置 pre-commit 钩子..."
cat <<'EOF' > "$PRE_COMMIT_FILE"
#!/bin/sh

# 设置最大文件大小（例如：50MB）
max_size=52428800  # 50MB，单位是字节

# 遍历所有将被提交的文件
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        file_size=$(stat -c %s "$file")
        if [ $file_size -gt $max_size ]; then
            echo "Error: File $file is larger than 50MB and will not be committed."
            exit 1
        fi
    fi
done

exit 0
EOF

# 设置 pre-commit 文件可执行权限
chmod +x "$PRE_COMMIT_FILE"
echo "pre-commit 钩子已配置完成！"

# 显示结果
echo "初始化完成！以下是项目当前状态："
git status
