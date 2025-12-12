curl https://vllm.salt-lab.org/v1/completions -H "Authorization: Bearer sk-SzGKL7KOo8fs3vkuFx5mHRdwZktp_0veWDek9YTIpGY" -H "Content-Type: application/json" -d '{
        "model": "openai/gpt-oss-20b",
        "messages": [
          {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
      }' | jq
