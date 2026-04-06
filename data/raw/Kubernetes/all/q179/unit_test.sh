kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/hostaliases-pod --timeout=20s
kubectl describe pod huge-pages-example | grep "Limits:
      hugepages-2Mi:  100Mi
      memory:         100Mi
    Requests:
      hugepages-2Mi:  100Mi
      memory:         100Mi" && echo cloudeval_unit_test_passed