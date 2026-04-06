# kubectl create secret docker-registry regcred \
#   --docker-server=https://index.docker.io/v1/ \
#   --docker-username=usr \
#   --docker-password="pwd" \
#   --docker-email="1@gmail.com"
# minikube addons enable registry
# echo "apiVersion: v1
# kind: Secret
# metadata:
#   name: regcred
# data:
#   .dockerconfigjson: eyJhdXRocyI6eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsidXNlcm5hbWUiOiJ1c3IiLCJwYXNzd29yZCI6InB3ZCIsImVtYWlsIjoiMUBnbWFpbC5jb20iLCJhdXRoIjoiZFhOeU9uQjNaQT09In19fQ==
# type: kubernetes.io/dockerconfigjson" | kubectl apply -f -
kubectl apply -f labeled_code.yaml
sleep 20
kubectl describe pod liveness-http | grep "Liveness probe failed: HTTP probe failed with statuscode: 500" && echo "Correct behavior, keep going" || { echo "Wrong behavior, stopping execution"; exit 1; }
sleep 10
kubectl describe pod liveness-http | grep "Killing" && echo cloudeval_unit_test_passed

