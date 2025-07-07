#!/bin/bash

# 1. 종료할 프로세스 키워드 입력받기
KEYWORD="$1"

if [ -z "$KEYWORD" ]; then
  echo "사용법: $0 <프로세스_이름_또는_키워드>"
  exit 1
fi

# 2. 현재 사용자
USER=$(whoami)

# 3. 대상 프로세스 PID 찾기
PIDS=$(ps -ef | grep "$KEYWORD" | grep "$USER" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "[$KEYWORD] 에 해당하는 실행 중인 프로세스를 찾을 수 없습니다."
else
  kill -9 $PIDS
  sleep 1  # 종료 반영 대기
fi

