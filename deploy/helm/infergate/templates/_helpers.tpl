{{- define "infergate.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "infergate.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{/* Model id as a DNS label, the {slug} of gpu.worker_url_template. */}}
{{- define "infergate.slug" -}}
{{- regexReplaceAll "[^a-z0-9-]" (lower .) "-" -}}
{{- end -}}

{{/*
Variables of the model YAMLs as JSON, the same for the gateway and every worker: the gateway resolves
the YAMLs too and sends its result to a worker on /reload. Every GPU worker gets its own <ID>_GPU
index (0, 1, ... in list order), so the gateway counts the VRAM of each card apart; models.env wins.
*/}}
{{- define "infergate.modelEnv" -}}
{{- $env := dict -}}
{{- $index := 0 -}}
{{- range .Values.workers -}}
{{- $gpus := 1 -}}
{{- if hasKey . "gpus" -}}{{- $gpus = int .gpus -}}{{- end -}}
{{- if gt $gpus 0 -}}
{{- $prefix := .envPrefix | default (upper (regexReplaceAll "[^A-Za-z0-9]" .id "_")) -}}
{{- $_ := set $env (printf "%s_GPU" $prefix) (toString $index) -}}
{{- $index = add1 $index -}}
{{- end -}}
{{- end -}}
{{- range $name, $value := .Values.models.env -}}
{{- $_ := set $env $name (toString $value) -}}
{{- end -}}
{{- toJson $env -}}
{{- end -}}

{{- define "infergate.modelsConfigMap" -}}
{{- if .Values.models.files -}}
{{- printf "%s-models" (include "infergate.fullname" .) -}}
{{- else -}}
{{- required "set models.existingConfigMap or models.files" .Values.models.existingConfigMap -}}
{{- end -}}
{{- end -}}
