kubectl apply -f labeled_code.yaml
sleep 5
kubectl wait --for=condition=initialized pod -l app=zk --timeout=60s

[ "$(kubectl get sts zk -o jsonpath='{.spec.replicas}')" = "3" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.template.spec.securityContext.runAsUser}')" = "1000" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.template.spec.securityContext.fsGroup}')" = "1000" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.volumeClaimTemplates[0].spec.resources.requests.storage}')" = "1Gi" ] && \
echo cloudeval_unit_test_passed
