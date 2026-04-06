kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=debug-agent --timeout=120s

pods=$(kubectl get pods -l app=debug-agent --output=jsonpath={.items..metadata.name})

kubectl logs $pods | grep "nginx" \
&& kubectl exec $pods -- ls /var/run/docker.sock \
&& kubectl get pod $pods -o=jsonpath='{.spec.containers[0].livenessProbe.httpGet.path}' | grep "/healthz" && echo cloudeval_unit_test_passed
