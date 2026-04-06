echo "apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
type: Opaque
data:
  MYSQL_USER: bXlzcWwK
  MYSQL_PASSWORD: bXlzcWwK
  MYSQL_DATABASE: c2FtcGxlCg==
  MYSQL_ROOT_PASSWORD: c3VwZXJzZWNyZXQK" | kubectl create -f -
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/mysql-pod --timeout=20s
kubectl describe pod mysql-pod | grep "Environment Variables from:
      mysql-secret  Secret" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/33478555/kubernetes-equivalent-of-env-file-in-docker