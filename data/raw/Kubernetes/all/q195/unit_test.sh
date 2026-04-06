minikube addons enable metrics-server
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=frontend-app --timeout=60s

CPU_REQUEST=$(kubectl get rs frontend -o=jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')
HPA_MIN_REPLICAS=$(kubectl get hpa frontend-scaler -o=jsonpath='{.spec.minReplicas}')
HPA_MAX_REPLICAS=$(kubectl get hpa frontend-scaler -o=jsonpath='{.spec.maxReplicas}')

[ "$CPU_REQUEST" = "200m" ] && \
[ "$HPA_MIN_REPLICAS" -eq 3 ] && \
[ "$HPA_MAX_REPLICAS" -eq 10 ] && \
echo cloudeval_unit_test_passed
