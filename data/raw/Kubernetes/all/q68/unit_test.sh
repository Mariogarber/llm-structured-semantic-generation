kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deployment/kettle --timeout=60s 
deployment_name=$(kubectl get deployment kettle -o=jsonpath='{.metadata.name}')
env_vars=$(kubectl get deployment kettle -o=jsonpath='{.spec.template.spec.containers[0].env}')

[ "$deployment_name" = "kettle" ] && \
echo "$env_vars" | grep -q "DEPLOYMENT" && \
echo "$env_vars" | grep -q "prod" && \
echo "$env_vars" | grep -q "SUBSCRIPTION_PATH" && \
echo "$env_vars" | grep -q "local-subscription" && \
echo cloudeval_unit_test_passed
