kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/hello-world --timeout=30s
kubectl get pods
kubectl logs hello-world | grep "hello world" && kubectl logs hello-world | grep "hello world again" && echo cloudeval_unit_test_passed
# stackoverflow https://stackoverflow.com/questions/33887194/how-to-set-multiple-commands-in-one-yaml-file-with-kubernetes