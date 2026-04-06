kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=initialized pod two-containers --timeout=60s
sleep 10

content=$(kubectl exec -it two-containers -c nginx -- /bin/sh -c "apt-get update > /dev/null && apt-get install -y curl > /dev/null && curl localhost")
echo $content | grep -q "Hello from busybox" && \
echo cloudeval_unit_test_passed