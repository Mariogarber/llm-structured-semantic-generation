echo "apiVersion: v1
kind: Namespace
metadata:
  name: namespace-a
  labels:
    name: namespace-a" | kubectl create -f -
echo "apiVersion: v1
kind: Namespace
metadata:
  name: namespace-b
  labels:
    name: namespace-b" | kubectl create -f -
kubectl apply -f labeled_code.yaml
sleep 10
echo "apiVersion: v1
kind: Pod
metadata:
  name: service-x-pod
  namespace: namespace-b
  labels:
    app: service-x
spec:
  containers:
  - name: nginx
    image: nginx:alpine
    ports:
    - containerPort: 80

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: service-x
  namespace: namespace-b
spec:
  selector:
    app: service-x
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80" | kubectl apply -f -
sleep 10
echo "apiVersion: v1
kind: Pod
metadata:
  name: client-pod
  namespace: namespace-a
spec:
  containers:
  - name: busybox
    image: busybox:1.28
    command:
      - sleep
      - \"3600\"" | kubectl apply -f -
sleep 20
kubectl exec -n namespace-a -it client-pod -- /bin/sh -c "wget -O- service-x && exit" | grep "Welcome to nginx!" && echo cloudeval_unit_test_passed
# from stackoverflow https://stackoverflow.com/questions/37221483/service-located-in-another-namespace
