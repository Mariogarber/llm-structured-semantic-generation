kubectl apply -f labeled_code.yaml
echo "apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test-container
    image: busybox
    command: [ '/bin/sh', '-c', 'cat /config/game.properties && echo' ]
    volumeMounts:
    - name: config-volume
      mountPath: /config
  volumes:
    - name: config-volume
      configMap:
        name: game-demo" | kubectl create -f -
kubectl wait --for=condition=running pods/test-pod --timeout=20s
kubectl logs test-pod | grep "enemy.types=aliens,monsters
player.maximum-lives=5" && echo cloudeval_unit_test_passed