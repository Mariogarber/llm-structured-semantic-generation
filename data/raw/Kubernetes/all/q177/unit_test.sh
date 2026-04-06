kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/lifecycle-demo --timeout=20s
kubectl exec -it lifecycle-demo -- /bin/bash -c "cat /usr/share/message && exit" | grep "Hello from the postStart handler" && echo cloudeval_unit_test_passed