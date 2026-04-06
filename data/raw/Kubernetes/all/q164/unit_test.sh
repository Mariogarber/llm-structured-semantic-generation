kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/security-context-demo-4 --timeout=60s
kubectl get pod security-context-demo-4
kubectl exec -it security-context-demo-4 -- sh -c "cd /proc/1 && cat status && exit" | grep "CapPrm:	00000000aa0435fb
CapEff:	00000000aa0435fb" && echo cloudeval_unit_test_passed
# bits 12 and 25 are set. Bit 12 is CAP_NET_ADMIN, and bit 25 is CAP_SYS_TIME. 
# See https://github.com/torvalds/linux/blob/master/include/uapi/linux/capability.h for definitions of the capability constants.
