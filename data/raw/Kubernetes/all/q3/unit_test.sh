kubectl apply -f labeled_code.yaml
echo "apiVersion: v1
kind: Pod
metadata:
  name: test-pod-2
spec:
  containers:
  - name: test-container
    image: busybox
    command: [ '/bin/sh', '-c', 'cat /config/cluster-name && echo' ]
    volumeMounts:
    - name: config-volume
      mountPath: /config
  volumes:
    - name: config-volume
      configMap:
        name: cluster-info" | kubectl create -f -
kubectl wait --for=condition=running pods/test-pod-2 --timeout=20s
kubectl logs test-pod-2 | grep "foo" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/38242062/how-to-get-kubernetes-cluster-name-from-k8s-api