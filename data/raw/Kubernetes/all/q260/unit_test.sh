kubectl apply -f labeled_code.yaml
echo "apiVersion: v1
kind: Secret
metadata:
  name: secret-sa-sample
  annotations:
    kubernetes.io/service-account.name: "sa-name"
type: kubernetes.io/service-account-token
data:
  extra: YmFyCg==" | kubectl create -f -
kubectl get secret secret-sa-sample -o jsonpath='{.data.extra}' | base64 --decode | grep "bar" && echo cloudeval_unit_test_passed