kubectl apply -f labeled_code.yaml
msg=$(kubectl get sa anno-sys-sa -n kube-system -o jsonpath='{.metadata.annotations.example\.com/description}')

echo $msg | grep -q "This is an annotated SA in kube-system" && \
kubectl get sa -n kube-system | grep -q "anno-sys-sa" && \
echo cloudeval_unit_test_passed
