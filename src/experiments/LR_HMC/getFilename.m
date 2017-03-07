function fn = getFilename(str)

dt = datestr(now);
if strcmp(dt(end-7),' ')
    hr = dt(end-6);
else
    hr = dt(end-7:end-6);
end

mn = dt(end-4:end-3);
sc = dt(end-1:end);
dt = strrep(datestr(date,26),'/','');
dt = dt(3:end);
fn = strcat(dt,'_',strcat(hr,mn,sc),'_',str);
