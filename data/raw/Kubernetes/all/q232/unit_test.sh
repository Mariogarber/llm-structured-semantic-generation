echo "apiVersion: v1
kind: ServiceAccount
metadata:
  name: sa-name" | kubectl create -f -
kubectl apply -f labeled_code.yaml
kubectl get secret secret-sa-sample -o jsonpath='{.data.extra}' | base64 --decode | grep "bar" && echo cloudeval_unit_test_passed