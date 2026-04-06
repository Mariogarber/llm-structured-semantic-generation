kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=ds-hostport --timeout=60s
minikube_ip=$(minikube ip)

curl_output=$(curl "$minikube_ip:30080")
if echo "$curl_output" | grep "Welcome to nginx!"; then
  echo cloudeval_unit_test_passed
fi