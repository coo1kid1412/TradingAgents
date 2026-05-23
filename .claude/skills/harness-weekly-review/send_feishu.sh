#!/bin/bash
# 发飞书自定义机器人 text 消息。
#
# 用法：send_feishu.sh "<消息正文>"
# 退出码：0 = 成功；非 0 = 失败（curl 错 / 飞书 code ≠ 0）

set -e

WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/57885d8e-cd29-4608-9769-0cfd7341a0ff"

MSG="$1"
if [ -z "$MSG" ]; then
    echo "用法: $0 '<消息正文>'" >&2
    exit 1
fi

# 用 python json.dumps 做转义，避免引号/换行炸 shell
PAYLOAD=$(python3 -c "
import sys, json
msg = sys.argv[1]
print(json.dumps({'msg_type': 'text', 'content': {'text': msg}}))
" "$MSG")

RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
    -d "$PAYLOAD" "$WEBHOOK")

echo "飞书响应: $RESPONSE"

# 飞书成功返回 {"code":0,"msg":"ok",...} 或 {"StatusCode":0,...}
if echo "$RESPONSE" | grep -qE '"code"\s*:\s*0|"StatusCode"\s*:\s*0'; then
    exit 0
else
    echo "飞书发送失败：$RESPONSE" >&2
    exit 1
fi
