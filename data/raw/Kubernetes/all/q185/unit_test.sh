minikube addons enable metrics-server
# kubectl create namespace mem-example
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
sleep 30
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/memory-demo --timeout=30s
kubectl describe pod memory-demo| grep "Limits:
      memory:  20Mi
    Requests:
      memory:     10Mi" && echo cloudeval_unit_test_passed
# INCLUDE: "15Mi"
