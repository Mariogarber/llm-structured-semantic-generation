kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-share-process-ns --timeout=60s
pods=$(kubectl get pods --selector=name=ds-share-process-ns --output=jsonpath={.items..metadata.name})
kubectl logs $pods -c container-sidecar | grep "Main container" && echo cloudeval_unit_test_passed


