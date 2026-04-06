kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready deployment --all --timeout=20s
kubectl describe deployment nginx-dep-correct | grep "nfs-pvc" && echo cloudeval_unit_test_passed
# Stackoverflow https://stackoverflow.com/questions/54479397/error-converting-yaml-to-json-did-not-find-expected-key-kubernetes