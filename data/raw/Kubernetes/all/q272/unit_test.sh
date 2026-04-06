kubectl apply -f labeled_code.yaml
sleep 10
kubectl wait --for=condition=ready pod -l app=nginx --timeout=60s

kubectl get statefulset web -o jsonpath='{.metadata.name}' | grep -q 'web' && \
[ "$(kubectl get statefulset web -o jsonpath='{.spec.replicas}')" -eq 2 ] && \
[ "$(kubectl get statefulset web -o jsonpath='{.spec.template.spec.containers[0].image}')" = "registry.k8s.io/nginx-slim:0.8" ] && \
kubectl get statefulset web -o jsonpath='{.spec.template.spec.containers[0].volumeMounts[0].mountPath}' | grep -q '/usr/share/nginx/html' && \
[ "$(kubectl get pvc www-web-0 -o jsonpath='{.spec.accessModes[0]}')" = "ReadWriteOnce" ] &&
[ "$(kubectl get pvc www-web-0 -o jsonpath='{.spec.resources.requests.storage}')" = "100Mi" ] && \
echo cloudeval_unit_test_passed
