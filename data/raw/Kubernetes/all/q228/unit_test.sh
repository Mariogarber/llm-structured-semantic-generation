kubectl apply -f labeled_code.yaml
echo "apiVersion: v1
kind: Pod
metadata:
  name: secret-dotfiles-pod
spec:
  volumes:
    - name: secret-volume
      secret:
        secretName: dotfile-secret
  containers:
    - name: dotfile-test-container
      image: registry.k8s.io/busybox
      command:
        - ls
        - \"-l\"
        - \"/etc/secret-volume\"
      volumeMounts:
        - name: secret-volume
          readOnly: true
          mountPath: \"/etc/secret-volume\"" | kubectl create -f -
kubectl wait --for=condition=complete pods/secret-dotfiles-pod --timeout=20s
kubectl describe pod secret-dotfiles-pod | grep "/etc/secret-volume from secret-volume" && kubectl describe pod secret-dotfiles-pod | grep "secret-volume:
    Type:        Secret" && kubectl get secret | grep "dotfile-secret" && echo cloudeval_unit_test_passed
