kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/hostaliases-pod --timeout=20s
kubectl logs hostaliases-pod | grep "# Entries added by HostAliases.
127.0.0.1	foo.local	bar.local
10.1.2.3	foo.remote	bar.remote" && echo cloudeval_unit_test_passed