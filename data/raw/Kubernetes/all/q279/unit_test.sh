kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/my-pod --timeout=60s

[ "$(kubectl get sc my-sc -o jsonpath='{.provisioner}')" = "k8s.io/minikube-hostpath" ] && \
[ "$(kubectl get pvc my-pvc -o jsonpath='{.spec.resources.requests.storage}')" = "1Gi" ] && \
[ "$(kubectl get pod my-pod -o jsonpath='{.spec.containers[0].image}')" = "nginx:latest" ] && \
[ "$(kubectl get pod my-pod -o jsonpath='{.spec.nodeSelector.kubernetes\.io/hostname}')" = "minikube" ] && \
[ "$(kubectl get pod my-pod -o jsonpath='{.spec.volumes[0].persistentVolumeClaim.claimName}')" = "my-pvc" ] && \
[ "$(kubectl get pod my-pod -o jsonpath='{.spec.containers[0].volumeMounts[0].mountPath}')" = "/usr/share/nginx/html" ] && \
echo cloudeval_unit_test_passed
