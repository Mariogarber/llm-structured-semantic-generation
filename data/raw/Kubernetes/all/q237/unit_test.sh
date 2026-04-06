kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod/secret-dotfiles-pod --timeout=60s

! kubectl logs pod/secret-dotfiles-pod | grep -q ".secret-file" && \
kubectl exec secret-dotfiles-pod -- ls -la /etc/secret-volume | grep -q ".secret-file" && \
kubectl exec secret-dotfiles-pod -- cat /etc/secret-volume/.secret-file | grep -q "value-2" && \
echo cloudeval_unit_test_passed
