kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/init-demo --timeout=40s
kubectl exec -it init-demo -- /bin/sh -c "apt-get update && apt-get install curl && curl localhost && exit" | grep "info.cern.ch" && echo cloudeval_unit_test_passed
# INCLUDE: "info.cern.ch"
