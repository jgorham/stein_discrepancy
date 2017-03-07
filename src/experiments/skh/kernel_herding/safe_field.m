function value = safe_field(struct,field_name,default_value)
% function safe_field(struct,field_name,default_value)
% return value of struct.field_name if it exists;
%        or default_value otherwise

if isfield(struct, field_name)
    value = struct.(field_name);
else
    value = default_value;
end