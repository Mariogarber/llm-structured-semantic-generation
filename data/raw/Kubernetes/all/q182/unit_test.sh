echo "apiVersion: v1
kind: Secret
metadata:
  name: dotfile-secret
data:
  .secret-file: dmFsdWUtMg0KDQo=" | kubectl create -f -
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/secret-dotfiles-pod --timeout=20s
kubectl describe pod secret-dotfiles-pod | grep "/etc/secret-volume from secret-volume" && kubectl describe pod secret-dotfiles-pod | grep "secret-volume:
    Type:        Secret" && echo cloudeval_unit_test_passed