kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=dnsconfig --timeout=20s
pods=$(kubectl get pods --selector=app=dnsconfig --output=jsonpath={.items..metadata.name})
kubectl exec  $pods -- cat /etc/resolv.conf | grep 'nameserver' | grep '8.8.8.8' && echo cloudeval_unit_test_passed1
kubectl exec  $pods -- cat /etc/resolv.conf | grep 'search' | grep 'mydomain.com' && echo cloudeval_unit_test_passed2