#!/bin/bash

source /opt/workshops/elastic-retry.sh

# Get Elastic Stack Version from Ansible template variable
ELASTIC_STACK_VERSION=9.4.2

model=claude-sonnet-4
connector=true
knowledgebase=true
docs=false
everywhere=true
prompt=false
name=claude-sonnet-4
while getopts "m:k:c:p:d:e:n:" opt
do
   case "$opt" in
      c ) connector="$OPTARG" ;;
      m ) model="$OPTARG" ;;
      k ) knowledgebase="$OPTARG" ;;
      d ) docs="$OPTARG" ;;
      e ) everywhere="$OPTARG" ;;
      p ) prompt="$OPTARG" ;;
      n ) name="$OPTARG" ;;
   esac
done
echo "model=$model"
echo "knowledgebase=$knowledgebase"
echo "docs=$docs"
echo "everywhere=$everywhere"
echo "prompt=$prompt"
echo "name=$name"

####################################################################### ENV

ENV_FILE_PARENT_DIR=/home/kubernetes-vm
ENV_FILE=$ENV_FILE_PARENT_DIR/env
export $(cat $ENV_FILE | xargs)

####################################################################### OPENAI
# Install LLM in ES

if [ "$connector" = true ] ; then
echo "Adding LLM connector"
add_connector() {
    local http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$KIBANA_URL_LOCAL/api/actions/connector" \
    -H 'Content-Type: application/json' \
    --header "kbn-xsrf: true" --header "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" --header "x-elastic-internal-origin: Kibana"  -d'
    {
    "name":"'"$name"'",
    "config": {
        "apiProvider":"OpenAI",
        "apiUrl":"https://'"$LLM_PROXY_URL"'/v1/chat/completions",
        "defaultModel": "'"$model"'"
    },
    "secrets": {
        "apiKey": "'"$LLM_APIKEY"'"
    },
    "connector_type_id":".gen-ai"
    }')

    if echo $http_status | grep -q '^2'; then
        echo "Connector added successfully with HTTP status: $http_status"
        return 0
    else
        echo "Failed to add connector. HTTP status: $http_status"
        return 1
    fi
}
retry_command_lin add_connector
fi # if [ "$connector" = true ]

if [ "$knowledgebase" = true ] ; then
# init knowledgebase
echo "Initializing knowledgebase"
init_kb() {
  local http_status

  if [[ $ELASTIC_STACK_VERSION == 8.* ]]; then
    http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$KIBANA_URL_LOCAL/internal/observability_ai_assistant/kb/setup" \
      -H 'Content-Type: application/json' \
      --header "kbn-xsrf: true" \
      --header "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" \
      --header 'x-elastic-internal-origin: Kibana')
  elif [[ $ELASTIC_STACK_VERSION == 9.* ]]; then
    http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$KIBANA_URL_LOCAL/internal/observability_ai_assistant/kb/setup?inference_id=.elser-2-elasticsearch" \
      -H 'Content-Type: application/json' \
      --header "kbn-xsrf: true" \
      --header "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" \
      --header 'x-elastic-internal-origin: Kibana')
  fi

  if [[ $http_status =~ ^2 ]]; then
    echo "Elastic knowledgebase successfully initialized: $http_status"
    return 0
  else
    echo "Failed to initialize Elastic knowledgebase. HTTP status: $http_status"
    return 1
  fi
}
retry_command_lin init_kb

wait_kb() {
    output=$(curl -X GET -s "$KIBANA_URL_LOCAL/internal/observability_ai_assistant/kb/status" \
        -H "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" \
        -H "Content-Type: application/json" \
        -H "kbn-xsrf: true" \
        -H 'x-elastic-internal-origin: Kibana')

    ENABLED=$(echo "$output" | jq -r '.enabled')

    if [[ $ELASTIC_STACK_VERSION == 8.* ]]; then
        READY=$(echo "$output" | jq -r '.ready')
        MODEL_DEPLOYMENT_STATE=$(echo "$output" | jq -r '.model_stats.deployment_state')
        MODEL_ALLOCATION_STATE=$(echo "$output" | jq -r '.model_stats.allocation_state')

    elif [[ $ELASTIC_STACK_VERSION == 9.1.* ]]; then
      KBSTATE=$(echo "$output" | jq -r '.kbState')

    elif [[ $ELASTIC_STACK_VERSION == 9.* ]]; then
        KBSTATE=$(echo "$output" | jq -r '.inferenceModelState')
    fi

    # Echo vars if they are available
    for var in READY ENABLED MODEL_DEPLOYMENT_STATE MODEL_ALLOCATION_STATE KBSTATE; do
        [[ -n "${!var}" && "${!var}" != "null" ]] && echo "$var: ${!var}"
    done

    if [[ $ENABLED == true && $ELASTIC_STACK_VERSION == 8.* && $READY == true && $MODEL_DEPLOYMENT_STATE == "started" && $MODEL_ALLOCATION_STATE == "fully_allocated" ]]; then
        echo "o11y kb is ready on $attempt"
        return 0
    elif [[ $ENABLED == true && $ELASTIC_STACK_VERSION == 9.* && $KBSTATE == "READY" ]]; then
        echo "o11y kb is ready on $attempt"
        return 0
    else
        echo "o11y kb is not ready on attempt $attempt: $output"
        return 1
    fi
}
retry_command_lin wait_kb
fi # if [ "$knowledgebase" = true ]

if [ "$docs" = true ] ; then
echo "Initializing Elastic documentation"
init_documentation() {
  local http_status

  if [[ $ELASTIC_STACK_VERSION == 8.* ]]; then
    http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$KIBANA_URL_LOCAL/internal/product_doc_base/install" \
      -H "Content-Type: application/json" \
      -H "kbn-xsrf: true" \
      -H "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" \
      -H "x-elastic-internal-origin: Kibana")
  
  elif [[ $ELASTIC_STACK_VERSION == 9.* ]]; then
    http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$KIBANA_URL_LOCAL/internal/product_doc_base/install" \
      -H "Content-Type: application/json" \
      -H "kbn-xsrf: true" \
      -H "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" \
      -H "x-elastic-internal-origin: Kibana" \
      -d '{"inferenceId":".elser-2-elasticsearch"}')
  fi

  if [[ $http_status =~ ^2 ]]; then
    echo "Elastic documentation successfully initialized: $http_status"
    return 0
  else
    echo "Failed to initialize Elastic documentation. HTTP status: $http_status"
    return 1
  fi
}
retry_command_lin init_documentation
fi # if [ "$docs" = true ]

if [ "$everywhere" = true ] ; then
echo "Initializing AI Assistant Everywhere"
init_ai_everywhere() {
    local http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$KIBANA_URL_LOCAL/internal/kibana/settings" \
    -H 'Content-Type: application/json'\
    --header "kbn-xsrf: true" --header "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" --header 'x-elastic-internal-origin: Kibana'\
    -d '{"changes":{"aiAssistant:preferredAIAssistantType":"observability"}}')

    if echo $http_status | grep -q '^2'; then
        echo "Elastic AI Assistant Everywhere successfully initialized: $http_status"
        return 0
    else
        echo "Failed to initialize Elastic AI Assistant Everywhere. HTTP status: $http_status"
        return 1
    fi
}
retry_command_lin init_ai_everywhere
fi #if [ "$everywhere" = true ]

if [ "$prompt" = true ] ; then
curl -X PUT "$KIBANA_URL_LOCAL/internal/observability_ai_assistant/kb/user_instructions" \
  --header 'Content-Type: application/json' \
  --header "kbn-xsrf: true" \
  --header "Authorization: Basic $ELASTICSEARCH_AUTH_BASE64" \
  --header 'x-elastic-internal-origin: Kibana' \
  -d @/opt/workshops/elastic-llm-prompt.json
fi #if [ "$prompt" = true ]
