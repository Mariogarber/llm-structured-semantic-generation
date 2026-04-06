kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=ds-hostpath --timeout=60s
pods=$(kubectl get pods --selector=app=ds-hostpath --output=jsonpath={.items[0]..metadata.name})
kubectl describe pod $pods | grep "/var/log/nginx" && echo cloudeval_unit_test_passed1
kubectl exec $pods -- ls -l /var/log/nginx && echo cloudeval_unit_test_passed2