kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/config-pod-1 --timeout=20s
kubectl describe pod config-pod-1 | grep "Environment:
      MY_POD_NAME:       config-pod-1 (v1:metadata.name)
      MY_POD_NAMESPACE:  default (v1:metadata.namespace)
      MY_POD_IP:          (v1:status.podIP)" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/30746888/how-to-know-a-pods-own-ip-address-from-inside-a-container-in-the-pod