kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=ds-hostpath --timeout=60s
kubectl exec -it $(kubectl get pods -l app=ds-hostpath -o=jsonpath='{.items[0].metadata.name}') -- ls /var/log/nginx && echo cloudeval_unit_test_passed
