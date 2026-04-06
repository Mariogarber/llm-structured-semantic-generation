kubectl apply -f labeled_code.yaml

kubectl create clusterrolebinding temp-binding --clusterrole=secret-creator-deletor --serviceaccount=default:default

kubectl create secret generic test-secret --from-literal=key=value

if kubectl get secret test-secret; then
  echo "secret passed"
else
  kubectl delete secret test-secret
  kubectl delete clusterrolebinding temp-binding
  kubectl apply -f labeled_code.yaml
  exit 1
fi

kubectl delete secret test-secret

if ! kubectl get secret test-secret 2>&1 | grep "not found"; then
  echo "Test failed"
else
  if kubectl get clusterrole secret-creator-deletor | grep "CREATED AT"; then
    echo cloudeval_unit_test_passed
  fi
fi

kubectl delete clusterrolebinding temp-binding
kubectl delete -f labeled_code.yaml
