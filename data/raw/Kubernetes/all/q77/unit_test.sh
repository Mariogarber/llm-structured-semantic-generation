kubectl label nodes minikube disktype=ssd
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deploy/nginx-deploy-2 --timeout=60s 
NODE_AFFINITY=$(kubectl get deployment nginx-deploy-2 -o=jsonpath='{.spec.template.spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key}')
NODE_AFFINITY_VALUE=$(kubectl get deployment nginx-deploy-2 -o=jsonpath='{.spec.template.spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]}')

kubectl label nodes minikube disktype-

[ "$NODE_AFFINITY" = "disktype" ] && \
[ "$NODE_AFFINITY_VALUE" = "ssd" ] && \
echo cloudeval_unit_test_passed
