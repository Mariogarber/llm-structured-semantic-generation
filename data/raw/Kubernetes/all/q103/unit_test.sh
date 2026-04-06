kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete job/example-job --timeout=60s

pods=$(kubectl get pods --selector=job-name=example-job -o=jsonpath='{.items[*].metadata.name}')

all_passed=true

for pod in $pods; do
  kubectl logs $pod | grep -q "Successfully pinged pod: example-job-0.headless-svc" && \
  kubectl logs $pod | grep -q "Successfully pinged pod: example-job-1.headless-svc" && \
  kubectl logs $pod | grep -q "Successfully pinged pod: example-job-2.headless-svc" || \
  all_passed=false
done

if kubectl get pods 2>&1 | grep "No resources found"; then
  all_passed=false
fi

if $all_passed; then
  echo cloudeval_unit_test_passed
else
  exit 1
fi
