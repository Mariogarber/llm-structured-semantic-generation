kubectl apply -f labeled_code.yaml
binding_info=$(kubectl get clusterrolebinding sample-clusterrolebinding -n kube-system -o yaml)

kubectl get clusterrolebinding -n kube-system | grep -q "sample-clusterrolebinding" && \
echo "$binding_info" | grep -q "name: sample-clusterrole" && \
echo "$binding_info" | grep -q "name: sample-sa" && \
echo "$binding_info" | grep -q "namespace: kube-system" && \
echo cloudeval_unit_test_passed
